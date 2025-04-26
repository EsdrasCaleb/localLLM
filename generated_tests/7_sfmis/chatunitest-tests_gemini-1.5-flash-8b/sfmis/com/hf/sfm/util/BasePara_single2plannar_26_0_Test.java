package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class BasePara_single2plannar_26_0_Test {

    @Test
    void single2plannar_nullParams() {
        BasePara basePara = new BasePara();
        String[][] result = basePara.single2plannar();
        assertNull(result);
    }

    @Test
    void single2plannar_evenParams() {
        BasePara basePara = new BasePara();
        String[] params = { "param1", "value1", "param2", "value2" };
        basePara.setQueryparams(params);
        String[][] result = basePara.single2plannar();
        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        assertEquals("param1", result[0][0]);
        assertEquals("value1", result[1][0]);
        assertEquals("param2", result[0][1]);
        assertEquals("value2", result[1][1]);
    }

    @Test
    void single2plannar_oddParams() {
        BasePara basePara = new BasePara();
        String[] params = { "param1", "value1", "param2" };
        basePara.setQueryparams(params);
        String[][] result = basePara.single2plannar();
        assertNull(result);
    }

    @Test
    void single2plannar_emptyParams() {
        BasePara basePara = new BasePara();
        String[] params = {};
        basePara.setQueryparams(params);
        String[][] result = basePara.single2plannar();
        assertNull(result);
    }
}
