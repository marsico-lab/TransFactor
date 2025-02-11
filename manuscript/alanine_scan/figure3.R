require(RColorBrewer)
require(gplots)
require(ggplot2)
require(ggrepel)
require(plotly)
require(gridExtra)

#set working dir
require(rstudioapi)
rstudioapi::getActiveDocumentContext
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

require(plyr)
require(dplyr)
require(tidyr)
require(stringr)
require(data.table)
require(gtools)

require(biomaRt)

ascan <- as.data.frame(fread("ascan.txt", header=TRUE, sep="\t"))
ascan.wt <- as.data.frame(fread("ascan.wt.txt", header=TRUE, sep="\t"))

ascan$window_size <- factor(ascan$window_size, levels = unique(ascan$window_size))

#PREDICTION VALUE BIASES
(panel_4a <- ggplot() +
  geom_density(data=ascan, aes(x = rel.seq, y = after_stat(..scaled..), group = window_size, color = window_size)) +
  scale_x_continuous(limits = c(-1, 1), breaks = seq(-1, 1, 0.25)) +
  ylab("density (scaled)") + xlab("relative prediction score [-0.25, 0.25] (sequence only)") +
  theme_classic())

ggsave(filename = ".\\finfig\\negative_prediction_bias.pdf" ,
       panel_4a,
       device = "pdf", width=8, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(panel_4a)


#domains - not in strong
dominfo <- as.data.frame(fread("Uniprot_Domain_Info_20127_genes.txt", header=TRUE, sep="\t"))
unique(dominfo$feature_type)

dom.filt <- dominfo %>% dplyr::filter(seq_id %in% ascan.wt$protein_id & feature_type %in% c("Domain", "Motif", "DNA binding")) %>%
  dplyr::mutate(len = end-start)

dom.filt.strong <- as.data.frame(table((dom.filt %>% dplyr::filter(seq_id %in% ascan.wt$protein_id))$attributes_clean))

dom.filt <- dom.filt %>%
  dplyr::filter(!(attributes_clean %in% dom.filt.strong$Var1[dom.filt.strong$Freq > 5]))

dom.filt <- dom.filt %>%
  rowwise() %>% do({
    temp <- .
    temp$position = c((temp$start-100):(temp$end+0))
    temp$rel.dom.pos = 1 + temp$position - temp$start
    as_tibble(temp)
  }) %>% ungroup() %>% as.data.frame() %>% dplyr::rename(protein_id = seq_id)

#remove negative rel.dom.pos that are part of another domain
dom.filt <- dom.filt %>% group_by(protein_id) %>% do({
  temp <- as.data.frame(.)
  temp %>% dplyr::filter(!(rel.dom.pos < 0 & position %in% temp$position[temp$rel.dom.pos >= 0]))
}) %>% ungroup() %>% as.data.frame()

pdomsx <- inner_join(dom.filt %>% dplyr::select(-source, -feature_type, -score, -strand, -phase),
                     ascan %>% dplyr::select(-position) %>% dplyr::rename(position = cor.pos) %>% dplyr::mutate(position = floor(position)), by = c("protein_id","position")) %>%
  mutate(npos = str_count(.$original_aa, pattern = "K") + str_count(.$original_aa, pattern = "R")) %>%
  mutate(naro = str_count(.$original_aa, pattern = "F") + str_count(.$original_aa, pattern = "Y") + str_count(.$original_aa, pattern = "W")) %>%
  group_by(protein_id, attributes_clean, window_size) %>% do({
    temp <- as.data.frame(.) %>% distinct(position, .keep_all = T)
  }) %>% ungroup() %>% as.data.frame()

pdomsx.p <- pdomsx %>% group_by(protein_id, window_size) %>% do({
  temp <- as.data.frame(.) %>% distinct(position, .keep_all = T)
}) %>% ungroup() %>% as.data.frame()

dom.arr.neg <- pdomsx.p %>% dplyr::filter(rel.dom.pos > 0)

median((pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct() %>% mutate(dlen = end - start))$dlen)

(tp <- ggplot() +
    geom_rect(aes(xmin = 0, xmax = 100, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.6) +
    geom_rect(aes(xmin = -as.numeric(as.character(unique(pdomsx.p$window_size)))/2, xmax = 0, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.1) +
    geom_line(data = pdomsx.p %>% group_by(rel.dom.pos, window_size) %>% mutate(rel.seq.med = median(rel.seq)) %>% dplyr::select(window_size, rel.dom.pos, rel.seq.med) %>% distinct(),
              aes(x=rel.dom.pos, y = rel.seq.med, group = window_size, color = window_size), linewidth = 2) +
    geom_hline(aes(yintercept = 0)) +
    geom_text(aes(x=-80, y = -0.08, label = paste0("N = ", nrow(pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct())))) +
    scale_x_continuous(limits = c(-100, 100), breaks = seq(-100,200,10)) +
    scale_y_continuous(limits = c(-0.1, 0.1), breaks = seq(-1,1,0.05)) +
    ylab("rel.seq.med") +
    theme_classic())

ggsave(filename = ".\\finfig\\domain_bias_notinstrongtwice.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

#DOMAINS - in strong
dominfo <- as.data.frame(fread("Uniprot_Domain_Info_20127_genes.txt", header=TRUE, sep="\t"))
unique(dominfo$feature_type)

dom.filt <- dominfo %>% dplyr::filter(seq_id %in% ascan.wt$protein_id & feature_type %in% c("Domain", "Motif", "DNA binding")) %>%
  dplyr::mutate(len = end-start)

dom.filt.strong <- as.data.frame(table((dom.filt %>% dplyr::filter(seq_id %in% ascan.wt$protein_id))$attributes_clean))

dom.filt <- dom.filt %>%
  dplyr::filter((attributes_clean %in% dom.filt.strong$Var1[dom.filt.strong$Freq > 5]))

dom.filt <- dom.filt %>%
  rowwise() %>% do({
    temp <- .
    temp$position = c((temp$start-100):(temp$end+0))
    temp$rel.dom.pos = 1 + temp$position - temp$start
    as_tibble(temp)
  }) %>% ungroup() %>% as.data.frame() %>% dplyr::rename(protein_id = seq_id)

#remove negative rel.dom.pos that are part of another domain
dom.filt <- dom.filt %>% group_by(protein_id) %>% do({
  temp <- as.data.frame(.)
  temp %>% dplyr::filter(!(rel.dom.pos < 0 & position %in% temp$position[temp$rel.dom.pos >= 0]))
}) %>% ungroup() %>% as.data.frame()

pdomsx <- inner_join(dom.filt %>% dplyr::select(-source, -feature_type, -score, -strand, -phase),
                     ascan %>% dplyr::select(-position) %>% dplyr::rename(position = cor.pos) %>% dplyr::mutate(position = floor(position)), by = c("protein_id","position")) %>%
  mutate(npos = str_count(.$original_aa, pattern = "K") + str_count(.$original_aa, pattern = "R")) %>%
  mutate(naro = str_count(.$original_aa, pattern = "F") + str_count(.$original_aa, pattern = "Y") + str_count(.$original_aa, pattern = "W")) %>%
  group_by(protein_id, attributes_clean, window_size) %>% do({
    temp <- as.data.frame(.) %>% distinct(position, .keep_all = T)
  }) %>% ungroup() %>% as.data.frame()

pdomsx.p <- pdomsx %>% group_by(protein_id, window_size) %>% do({
  temp <- as.data.frame(.) %>% distinct(position, .keep_all = T)
}) %>% ungroup() %>% as.data.frame()

dom.arr.pos <- pdomsx.p %>% dplyr::filter(rel.dom.pos > 0)
  
median((pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct() %>% mutate(dlen = end - start))$dlen)
  
(tp <- ggplot() +
  geom_rect(aes(xmin = 0, xmax = 100, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.6) +
  geom_rect(aes(xmin = -as.numeric(as.character(unique(pdomsx.p$window_size)))/2, xmax = 0, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.1) +
  
  geom_line(data = pdomsx.p %>% group_by(rel.dom.pos, window_size) %>% mutate(rel.seq.med = median(rel.seq)) %>% dplyr::select(window_size, rel.dom.pos, rel.seq.med) %>% distinct(),
            aes(x=rel.dom.pos, y = rel.seq.med, group = window_size, color = window_size), linewidth = 2) +
  geom_hline(aes(yintercept = 0)) +
  geom_text(aes(x=-80, y = -0.08, label = paste0("N = ", nrow(pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct())))) +
  scale_x_continuous(limits = c(-100, 100), breaks = seq(-100,200,10)) +
  scale_y_continuous(limits = c(-0.1, 0.1), breaks = seq(-1,1,0.05)) +
  ylab("rel.seq.med") +
  theme_classic())

ggsave(filename = ".\\finfig\\domain_bias.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

#Across domains
dom.arr.all <- rbind(dom.arr.pos %>% dplyr::mutate(tid = paste0("p_", window_size), sid = "p"),
                     dom.arr.neg %>% dplyr::mutate(tid = paste0("n_", window_size), sid = "n"))
dom.arr.all.stats <- dom.arr.all %>% group_by(tid, window_size) %>% do({
                       temp <- as.data.frame(.)
                       tibble(p = wilcox.test(temp$rel.seq, (ascan %>% dplyr::filter(window_size == unique(temp$window_size)))$rel.seq, alternative = "less")$p.value)
                     }) %>% ungroup() %>% as.data.frame()
dom.arr.all.stats <- dom.arr.all.stats %>% dplyr::mutate(p.adj = p.adjust(p, method = "fdr"))

dom.arr.all <- rbind(dom.arr.all %>% dplyr::select(protein_id, rel.seq, window_size, tid, sid),
                     ascan %>% dplyr::select(protein_id, rel.seq, window_size) %>% dplyr::mutate(tid = paste0("a_", window_size), sid = "a"))
(tp <- ggplot() +
  geom_boxplot(data = dom.arr.all, aes(x = window_size, y = rel.seq, group = tid, color = sid), outlier.shape = NA) +
  scale_y_continuous(limits = c(-0.2,0.2), breaks = seq(-0.2,0.2,0.1)) +
  theme_classic())
ggsave(filename = ".\\finfig\\domcomp.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)

(tp <- ggplot() +
    geom_boxplot(data = dom.arr.all, aes(x = window_size, y = rel.seq, group = tid, color = sid), outlier.shape = NA) +
    scale_y_continuous(limits = c(-0.15,0.15), breaks = seq(-0.2,0.2,0.1)) +
    theme_classic())
ggsave(filename = ".\\finfig\\domcomp.min.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)

#INDIVIDUAL DOMAINS
pdomsx.p <- pdomsx %>% dplyr::filter(attributes_clean == "PX")
median((pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct() %>% mutate(dlen = end - start))$dlen)
(tp <- ggplot() +
    geom_rect(aes(xmin = 0, xmax = 100, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.6) +
    geom_rect(aes(xmin = -as.numeric(as.character(unique(pdomsx.p$window_size)))/2, xmax = 0, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.1) +
    #geom_violin(data = pdomsx.p, aes(x=rel.dom.pos, y = rel.seq, group = rel.dom.pos)) +
    geom_line(data = pdomsx.p %>% group_by(rel.dom.pos, window_size) %>% mutate(rel.seq.med = median(rel.seq)) %>% dplyr::select(window_size, rel.dom.pos, rel.seq.med) %>% distinct(),
              aes(x=rel.dom.pos, y = rel.seq.med, group = window_size, color = window_size), linewidth = 2) +
    # geom_line(data = pdomsx.p %>% group_by(rel.dom.pos) %>% mutate(rel.seq.med = mean(rel.seq)) %>% dplyr::select(rel.dom.pos, rel.seq.med) %>% distinct(),
    #           aes(x=rel.dom.pos, y = rel.seq.med), color = "blue", linewidth = 2) +
    geom_hline(aes(yintercept = 0)) +
    geom_text(aes(x=-80, y = -0.08, label = paste0("N = ", nrow(pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct())))) +
    scale_x_continuous(limits = c(-100, 100), breaks = seq(-100,200,10)) +
    #scale_y_continuous(limits = c(-0.1, 0.1), breaks = seq(-1,1,0.05)) +
    ylab("rel.seq.med") +
    theme_classic())
ggsave(filename = ".\\finfig\\PX.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

unique(pdomsx$attributes_clean)
pdomsx.p <- pdomsx %>% dplyr::filter(attributes_clean == "COMM")
median((pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct() %>% mutate(dlen = end - start))$dlen)
(tp <- ggplot() +
  geom_rect(aes(xmin = 0, xmax = 100, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.6) +
  geom_rect(aes(xmin = -as.numeric(as.character(unique(pdomsx.p$window_size)))/2, xmax = 0, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.1) +
  #geom_violin(data = pdomsx.p, aes(x=rel.dom.pos, y = rel.seq, group = rel.dom.pos)) +
  geom_line(data = pdomsx.p %>% group_by(rel.dom.pos, window_size) %>% mutate(rel.seq.med = median(rel.seq)) %>% dplyr::select(window_size, rel.dom.pos, rel.seq.med) %>% distinct(),
            aes(x=rel.dom.pos, y = rel.seq.med, group = window_size, color = window_size), linewidth = 2) +
  # geom_line(data = pdomsx.p %>% group_by(rel.dom.pos) %>% mutate(rel.seq.med = mean(rel.seq)) %>% dplyr::select(rel.dom.pos, rel.seq.med) %>% distinct(),
  #           aes(x=rel.dom.pos, y = rel.seq.med), color = "blue", linewidth = 2) +
  geom_hline(aes(yintercept = 0)) +
  geom_text(aes(x=-80, y = -0.08, label = paste0("N = ", nrow(pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct())))) +
  scale_x_continuous(limits = c(-100, 100), breaks = seq(-100,200,10)) +
  #scale_y_continuous(limits = c(-0.1, 0.1), breaks = seq(-1,1,0.05)) +
  ylab("rel.seq.med") +
  theme_classic())
ggsave(filename = ".\\finfig\\COMM.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

pdomsx.p <- pdomsx %>% dplyr::filter(attributes_clean == "RRM")
median((pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct() %>% mutate(dlen = end - start))$dlen)
(tp <- ggplot() +
    geom_rect(aes(xmin = 0, xmax = 100, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.6) +
    geom_rect(aes(xmin = -as.numeric(as.character(unique(pdomsx.p$window_size)))/2, xmax = 0, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.1) +
    #geom_violin(data = pdomsx.p, aes(x=rel.dom.pos, y = rel.seq, group = rel.dom.pos)) +
    geom_line(data = pdomsx.p %>% group_by(rel.dom.pos, window_size) %>% mutate(rel.seq.med = median(rel.seq)) %>% dplyr::select(window_size, rel.dom.pos, rel.seq.med) %>% distinct(),
              aes(x=rel.dom.pos, y = rel.seq.med, group = window_size, color = window_size), linewidth = 2) +
    # geom_line(data = pdomsx.p %>% group_by(rel.dom.pos) %>% mutate(rel.seq.med = mean(rel.seq)) %>% dplyr::select(rel.dom.pos, rel.seq.med) %>% distinct(),
    #           aes(x=rel.dom.pos, y = rel.seq.med), color = "blue", linewidth = 2) +
    geom_hline(aes(yintercept = 0)) +
    geom_text(aes(x=-80, y = -0.08, label = paste0("N = ", nrow(pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct())))) +
    scale_x_continuous(limits = c(-100, 100), breaks = seq(-100,200,10)) +
    #scale_y_continuous(limits = c(-0.1, 0.1), breaks = seq(-1,1,0.05)) +
    ylab("rel.seq.med") +
    theme_classic())
ggsave(filename = ".\\finfig\\RRM.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

pdomsx.p <- pdomsx %>% dplyr::filter(attributes_clean == "KH 1")
median((pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct() %>% mutate(dlen = end - start))$dlen)
(tp <- ggplot() +
    geom_rect(aes(xmin = 0, xmax = 100, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.6) +
    geom_rect(aes(xmin = -as.numeric(as.character(unique(pdomsx.p$window_size)))/2, xmax = 0, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.1) +
    #geom_violin(data = pdomsx.p, aes(x=rel.dom.pos, y = rel.seq, group = rel.dom.pos)) +
    geom_line(data = pdomsx.p %>% group_by(rel.dom.pos, window_size) %>% mutate(rel.seq.med = median(rel.seq)) %>% dplyr::select(window_size, rel.dom.pos, rel.seq.med) %>% distinct(),
              aes(x=rel.dom.pos, y = rel.seq.med, group = window_size, color = window_size), linewidth = 2) +
    # geom_line(data = pdomsx.p %>% group_by(rel.dom.pos) %>% mutate(rel.seq.med = mean(rel.seq)) %>% dplyr::select(rel.dom.pos, rel.seq.med) %>% distinct(),
    #           aes(x=rel.dom.pos, y = rel.seq.med), color = "blue", linewidth = 2) +
    geom_hline(aes(yintercept = 0)) +
    geom_text(aes(x=-80, y = -0.08, label = paste0("N = ", nrow(pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct())))) +
    scale_x_continuous(limits = c(-100, 100), breaks = seq(-100,200,10)) +
    #scale_y_continuous(limits = c(-0.1, 0.1), breaks = seq(-1,1,0.05)) +
    ylab("rel.seq.med") +
    theme_classic())
ggsave(filename = ".\\finfig\\KH 1.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

pdomsx.p <- pdomsx %>% dplyr::filter(attributes_clean == "Helicase C-terminal")
median((pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct() %>% mutate(dlen = end - start))$dlen)
(tp <- ggplot() +
    geom_rect(aes(xmin = 0, xmax = 100, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.6) +
    geom_rect(aes(xmin = -as.numeric(as.character(unique(pdomsx.p$window_size)))/2, xmax = 0, ymin = -0.1, ymax = 0.1), fill = "coral1", alpha = 0.1) +
    #geom_violin(data = pdomsx.p, aes(x=rel.dom.pos, y = rel.seq, group = rel.dom.pos)) +
    geom_line(data = pdomsx.p %>% group_by(rel.dom.pos, window_size) %>% mutate(rel.seq.med = median(rel.seq)) %>% dplyr::select(window_size, rel.dom.pos, rel.seq.med) %>% distinct(),
              aes(x=rel.dom.pos, y = rel.seq.med, group = window_size, color = window_size), linewidth = 2) +
    # geom_line(data = pdomsx.p %>% group_by(rel.dom.pos) %>% mutate(rel.seq.med = mean(rel.seq)) %>% dplyr::select(rel.dom.pos, rel.seq.med) %>% distinct(),
    #           aes(x=rel.dom.pos, y = rel.seq.med), color = "blue", linewidth = 2) +
    geom_hline(aes(yintercept = 0)) +
    geom_text(aes(x=-80, y = -0.08, label = paste0("N = ", nrow(pdomsx.p %>% dplyr::select(protein_id, attributes_clean, start, end) %>% distinct())))) +
    scale_x_continuous(limits = c(-100, 200), breaks = seq(-100,200,10)) +
    #scale_y_continuous(limits = c(-0.1, 0.1), breaks = seq(-1,1,0.05)) +
    ylab("rel.seq.med") +
    theme_classic())
ggsave(filename = ".\\finfig\\Helicase C-terminal.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

pdomsx.stat <- pdomsx %>% dplyr::filter(rel.dom.pos > 0 & rel.dom.pos <= len & !is.na(attributes_clean)) %>% group_by(protein_id, start, end, window_size, attributes_clean) %>% do({
  temp <- as.data.frame(.)
  tibble(meanyhat = mean(temp$rel.seq),
         medyhat = median(temp$rel.seq),
         minyhat = min(temp$rel.seq),
         ivs = paste0(temp$rel.seq, collapse = ";"))
}) %>% ungroup() %>% as.data.frame()

#Removing some very short domains
pdomsx.stat <- pdomsx.stat %>% dplyr::filter(attributes_clean %in% (pdomsx.stat %>% dplyr::filter(window_size == 40))$attributes_clean)

pdomsx.stat.m <- pdomsx.stat %>% dplyr::filter(window_size == 40) %>% group_by(attributes_clean, window_size) %>% do({
  temp <- as.data.frame(.)
  
  tibble(mean.meanyhat = mean(temp$meanyhat),
         median.medianyhat = median(temp$medyhat),
         mean.minyhat = mean(temp$minyhat),
         n = nrow(temp),
         p = wilcox.test(x = unlist(sapply(temp$ivs, function(x){sapply(unlist(strsplit(x,";",T)[[1]]),as.numeric)})),
                         y = ascan$rel.seq[ascan$window_size == unique(temp$window_size)],
                         alternative = "less")$p.value)
}) %>% ungroup() %>% as.data.frame()

pdomsx.stat.m$p.adj <- p.adjust(pdomsx.stat.m$p, method = "fdr")


dorder <- pdomsx.stat.m %>% dplyr::filter(window_size == 40)
dorder <- dorder[order(dorder$mean.meanyhat, decreasing = F),]
pdomsx.stat$window_size <- factor(pdomsx.stat$window_size, levels = c(1, 5, 10, 15, 20, 25, 30, 35, 40))

pdomsx.stat$attributes_clean <- factor(pdomsx.stat$attributes_clean, levels = unique(dorder$attributes_clean))
pdomsx.stat.m$attributes_clean <- factor(pdomsx.stat.m$attributes_clean, levels = unique(dorder$attributes_clean))

pdomsx.stat <- pdomsx.stat[order(pdomsx.stat$attributes_clean),]
pdomsx.stat <- pdomsx.stat[order(pdomsx.stat$window_size),]
pdomsx.stat <- pdomsx.stat %>% mutate(grv = paste0(attributes_clean,"_",window_size))
pdomsx.stat$grv <- factor(pdomsx.stat$grv, levels = unique(pdomsx.stat$grv))


(tp <- ggplot() +
  geom_hline(aes(yintercept = 0)) +
  geom_boxplot(data=pdomsx.stat, aes(x=attributes_clean, y = meanyhat, group = grv, color = window_size), outlier.shape = NA) +
  theme_classic() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)))

ggsave(filename = ".\\finfig\\across_domains.pdf" ,
       tp,
       device = "pdf", width=18, height=6, units = "in", useDingbats=F, limitsize = FALSE)

goi = "COMMD4"
(tp <- ggplot() +
    geom_line(data = ascan %>% dplyr::filter(gene == goi),
              aes(x=cor.pos, y = seq, group = window_size, color = window_size), linewidth = 2) +
    geom_hline(data = ascan.wt %>% dplyr::filter(gene == goi), aes(yintercept = seq)) +
    scale_x_continuous(limits = c(0, (ascan.wt %>% dplyr::filter(gene == goi))$seq_len), breaks = c(seq(0, (ascan.wt %>% dplyr::filter(gene == goi))$seq_len, 10),(ascan.wt %>% dplyr::filter(gene == goi))$seq_len)) +
    xlab(paste0("position (", goi, ")")) +
    theme_classic())
ggsave(filename = ".\\finfig\\COMMD4.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)



clo <- as.data.frame(fread(".\\COMMs_clustalomega.txt", header=TRUE, sep="\t")) #Importing Clustal Omega aligned sequences
require(bio3d)

for(useq in clo$aaseq){
  if(useq==clo$aaseq[1]){
    seq.mat <- matrix(unlist(strsplit(useq,"")), nrow = 1, byrow = T)
  } else {
    seq.mat = rbind(seq.mat, matrix(unlist(strsplit(useq,"")), nrow = 1, byrow = T))
  }
}

seq.ent <- bio3d::entropy(seq.mat)

tp <- ggplot() +
  geom_line(aes(x=c(1:270), y=seq.ent$H.10.norm)) +
  scale_x_continuous(limits = c(234,265), breaks = seq(230,270,1)) +
  theme_classic()

ggsave(filename = ".\\finfig\\COMMD_alignment.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

(tp <- ggplot() +
  geom_line(aes(x=c(1:270), y=2+seq.ent$H.10.norm), color = "black") +
    geom_line(data = ascan.comms %>% dplyr::filter(window_size == 10), aes(x=c.cor.pos, y=seq, group = gene, color = gene)) +
  scale_x_continuous(limits = c(0, 270), breaks = seq(0,270,10)) +
  theme_classic())

ggsave(filename = ".\\finfig\\COMMD_alignment.full.pdf" ,
       tp,
       device = "pdf", width=6, height=3, units = "in", useDingbats=F, limitsize = FALSE)
rm(tp)

ggplot() +
  geom_line(aes(x=c(1:270), y=seq.ent$H.10.norm)) +
  #scale_x_continuous(limits = c(234,265), breaks = seq(230,270,10)) +
  theme_classic()


COMM_genes <- c(paste0("COMMD",c(1:10)),
                "CCDC22", "CCDC93",
                "VPS35L","VPS26C", "VPS29")

commseq <- as.data.frame(fread(".\\COMMDs_fasta.txt", header=TRUE, sep="\t")) %>% dplyr::rename(seq_id = uniprot_id)
commdom <- dominfo %>% dplyr::filter(attributes_clean == "COMM" & feature_type == "Domain")
comms <- inner_join(commseq, commdom, by = "seq_id")

commc <- as.data.frame(fread(".\\COMMs_clustalomega.txt", header=TRUE, sep="\t"))
add.comms <- as.data.frame(fread(".\\export\\add.comms.txt", header=TRUE, sep="\t"))

add.comms$window_size <- factor(add.comms$window_size, levels = unique(add.comms$window_size))

ascan.comms <- rbind(ascan %>% dplyr::filter(gene %in% commc$gene) %>% dplyr::select(gene, protein_id, window_size, position, cor.pos, seq, rel.seq, z.seq),
                     add.comms %>% dplyr::filter(gene %in% commc$gene) %>% dplyr::select(gene, protein_id, window_size, position, cor.pos, seq, rel.seq, z.seq))
ascan.comms$c.position <- ascan.comms$position
ascan.comms$c.cor.pos <- ascan.comms$cor.pos
for(ug in commc$gene){
  print(ug)
  for(i in c(1:270)){
    if(substr(commc$aaseq[commc$gene == ug],i,i)=="-"){
      ascan.comms$c.position[ascan.comms$gene == ug & ascan.comms$c.position >= i] <- ascan.comms$c.position[ascan.comms$gene == ug & ascan.comms$c.position >= i]+1
      ascan.comms$c.cor.pos[ascan.comms$gene == ug & ascan.comms$c.cor.pos >= i] <- ascan.comms$c.cor.pos[ascan.comms$gene == ug & ascan.comms$c.cor.pos >= i]+1
    }
  }
}

(tp <- ggplot() +
  geom_line(data = ascan.comms %>% dplyr::filter(window_size == 10 & c.cor.pos >= 234 & c.cor.pos <= 265 ), aes(x=c.cor.pos, y=seq, group = gene, color = gene)) +
  scale_x_continuous(limits = c(234, 265), breaks = seq(234,265,1)) +
  theme_classic())
ggsave(filename = ".\\finfig\\COMMDs_clustalomega.pdf" ,
       tp,
       device = "pdf", width=4, height=2, units = "in", useDingbats=F, limitsize = FALSE)



