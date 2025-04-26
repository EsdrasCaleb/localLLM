package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BasePara_single2plannar_26_2_Test {

    @Test
    public void testSingle2Planar() {
        // Arrange
        BasePara para = new BasePara();
        String[][] planarArr = para.single2plannar();
        // Act
        assertEquals(planarArr, new String[][] { { "param1", "param2" }, { "param3", "param4" } });
        // Assert
        assertTrue(planarArr[0][0] == "param1");
        assertTrue(planarArr[1][0] == "param2");
        assertTrue(planarArr[0][1] == "param3");
        assertTrue(planarArr[1][1] == "param4");
    }
}
