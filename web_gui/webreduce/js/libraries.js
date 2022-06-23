import 'https://unpkg.com/d3@5.16.0/dist/d3.min.js';
let d3 = window.d3;
export {d3};

import * as zip from 'https://cdn.jsdelivr.net/npm/@zip.js/zip.js@2.3.19/lib/zip-full.js';
export {zip};

export {default as Tree} from 'https://cdn.jsdelivr.net/gh/bmaranville/treejs@latest/src/index.js';
//export {default as Tree} from './treejs/src/index.js';

import { default as Vue } from 'https://cdn.jsdelivr.net/npm/vue@2.6.11/dist/vue.esm.browser.js';
import { default as VueMaterial } from 'https://cdn.skypack.dev/vue-material@latest';
Vue.use(VueMaterial);
//import { default as VueSimpleContextMenu } from 'https://cdn.skypack.dev/vue-simple-context-menu@^3.1.10';
//Vue.use(VueSimpleContextMenu);
//import { default as VueCustomContextMenu } from 'https://cdn.skypack.dev/vue-custom-context-menu@^3.0.2';
//Vue.use(VueCustomContextMenu);

// vuedraggable is available as a single-file esm source, Vue.Draggable/src/vuedraggable.js
// which depends only on Sortable, which can be found at 
// https://github.com/SortableJS/sortablejs/blob/master/modular/sortable.complete.esm.js
export { default as vuedraggable } from 'https://cdn.skypack.dev/vuedraggable';
export { Vue }

import * as messagepack from "https://unpkg.com/messagepack@1.1.11/dist/messagepack.es.js";
export { messagepack };
// PouchDB can be replaced with idb from https://github.com/jakearchibald/idb 
// since we are not using the sync features of pouch
import * as idb from 'https://unpkg.com/idb?module';
export { idb };
export {
    xyChart,
    heatChart,
    heatChartMultiMasked,
    colormap_names,
    get_colormap,
    dataflowEditor,
    extend,
    rectangleSelect,
    xSliceInteractor,
    ySliceInteractor,
    rectangleInteractor,
    ellipseInteractor,
    angleSliceInteractor,
    monotonicFunctionInteractor,
    scaleInteractor,
    rectangleSelectPoints
}  from 'https://cdn.jsdelivr.net/gh/usnistgov/d3-science@0.2.14/src/index.js';
//}  from './d3-science/src/index.js';

export { default as Split } from "https://cdn.skypack.dev/split.js";
export {default as json_patch} from "https://cdn.skypack.dev/json-patch-es6";
