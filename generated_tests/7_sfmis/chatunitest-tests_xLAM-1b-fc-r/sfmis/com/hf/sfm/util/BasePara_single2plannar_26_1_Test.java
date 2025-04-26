package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BasePara_single2plannar_26_1_Test {

    @Test
    public void testSingle2plannar() {
        BasePara basePara = new BasePara();
        String[] queryparams = { "param1", "param2", "param3", "param4" };
        basePara.setQueryparams(queryparams);
        String[][] expected = { { "param1", "param3" }, { "param2", "param4" } };
        assertEquals(expected, basePara.single2plannar());
        queryparams = new String[] { "param1", "param2", "param3", "param4", "param5", "param6" };
        basePara.setQueryparams(queryparams);
        expected = new String[][] { { "param1", "param3" }, { "param2", "param4" }, { "param5", "param6" } };
        assertEquals(expected, basePara.single2plannar());
    }
}
