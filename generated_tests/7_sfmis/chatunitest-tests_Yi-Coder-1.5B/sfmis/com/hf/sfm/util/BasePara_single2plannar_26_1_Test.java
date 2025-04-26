package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class BasePara_single2plannar_26_1_Test {

    @Test
    public void testSingle2plannar() {
        String[][] planarArr = new String[2][3];
        planarArr[0][0] = "a";
        planarArr[0][1] = "b";
        planarArr[0][2] = "c";
        planarArr[1][0] = "a1";
        planarArr[1][1] = "b1";
        planarArr[1][2] = "c1";
        BasePara basePara = new BasePara();
        basePara.setQueryparams(new String[] { "a", "b", "c" });
        basePara.setQueryparams(new String[] { "a1", "b1", "c1" });
        assertEquals(planarArr, basePara.single2plannar());
    }
}
