import { defineUserConfig } from "vuepress";
import theme from "./theme.js";

export default defineUserConfig({
  base: "/",

  lang: "zh-CN",
  title: "FH的博客",
  description: "FH的博客",

  theme,

  // Enable it with pwa
  // shouldPrefetch: false,
});
