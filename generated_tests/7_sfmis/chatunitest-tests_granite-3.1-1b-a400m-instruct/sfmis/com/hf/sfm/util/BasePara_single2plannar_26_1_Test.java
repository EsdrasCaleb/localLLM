package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BasePara_single2plannar_26_1_Test {

    @Test
    public void testSingle2Plannar() {
        BasePara basePara = new BasePara();
        String[][] planarArr = basePara.single2plannar();
        assertNotNull(planarArr);
        assertEquals(2, planarArr.length);
        assertEquals(2, planarArr[0].length);
        assertEquals("param1", planarArr[0][0]);
        assertEquals("param2", planarArr[0][1]);
        assertEquals("param3", planarArr[1][0]);
        assertEquals("param4", planarArr[1][1]);
    }
}
