package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class BasePara_single2plannar_26_0_Test {

    @Mock
    private BasePara basePara;

    @InjectMocks
    private BasePara testBasePara;

    @Test
    public void testSingle2Plannar_EmptyQueryParams_ReturnsNull() {
        when(basePara.getQueryparams()).thenReturn(null);
        String[][] result = testBasePara.single2plannar();
        assertEquals(null, result);
    }

    @Test
    public void testSingle2Plannar_OddLengthQueryParams_ThrowsException() {
        String[] queryparams = { "param1", "param2" };
        when(basePara.getQueryparams()).thenReturn(queryparams);
        assertThrows(OddParamsOfArrayInLoader.class, () -> testBasePara.single2plannar());
    }

    @Test
    public void testSingle2Plannar_EvenLengthQueryParams_ReturnsCorrectArray() {
        String[] queryparams = { "param1", "param2", "param3", "param4" };
        when(basePara.getQueryparams()).thenReturn(queryparams);
        String[][] result = testBasePara.single2plannar();
        String[][] expected = { { "param1", "param2" }, { "param3", "param4" } };
        assertEquals(expected, result);
    }

    @Test
    public void testSingle2Plannar_InvalidQueryParams_ThrowsException() {
        String[] queryparams = { "param1", "param2", "param3" };
        when(basePara.getQueryparams()).thenReturn(queryparams);
        assertThrows(OddParamsOfArrayInLoader.class, () -> testBasePara.single2plannar());
    }
}
