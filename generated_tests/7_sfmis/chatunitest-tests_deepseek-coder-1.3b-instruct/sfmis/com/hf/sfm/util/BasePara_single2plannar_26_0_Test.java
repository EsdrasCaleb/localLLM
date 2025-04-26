package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BasePara_single2plannar_26_0_Test {

    @Test
    public void testSingle2Plannar() {
        BasePara basePara = new BasePara();
        String[] queryParams = { "param1", "param2", "param3", "param4" };
        basePara.setQueryparams(queryParams);
        String[][] expectedResult = { { "param1", "param3" }, { "param2", "param4" } };
        String[][] actualResult = basePara.single2plannar();
        Assertions.assertArrayEquals(expectedResult, actualResult);
    }
}
