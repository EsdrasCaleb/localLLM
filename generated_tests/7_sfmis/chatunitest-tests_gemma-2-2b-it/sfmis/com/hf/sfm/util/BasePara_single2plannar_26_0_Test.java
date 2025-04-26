package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BasePara_single2plannar_26_0_Test {

    @Test
    void testSingle2Plannar() {
        BasePara basePara = new BasePara();
        String[][] planarArr = basePara.single2plannar();
        assertEquals(null, planarArr);
    }
}
