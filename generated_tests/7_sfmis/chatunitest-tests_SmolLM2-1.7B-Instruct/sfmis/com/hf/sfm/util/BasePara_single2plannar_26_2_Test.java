// Test method
package com.hf.sfm.util;

import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;

public class BasePara_single2plannar_26_2_Test {

    @Test
    public void testSingle2planar() {
        // Arrange
        BasePara basePara = mock(BasePara.class);
        String[] queryparams = { "param1", "param2" };
        when(basePara.getQueryparams()).thenReturn(queryparams);
        when(basePara.getArr2params()).thenReturn(new String[][] { { "param1", "param2" }, { "param3", "param4" } });
        when(basePara.getQuerySql()).thenReturn("select * from table where param1=? and param2=?");
        // Act
        String[][] result = basePara.single2plannar();
        // Assert
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals("param1", result[0][0]);
        assertEquals("param2", result[0][1]);
        assertEquals("param3", result[1][0]);
        assertEquals("param4", result[1][1]);
    }
}
