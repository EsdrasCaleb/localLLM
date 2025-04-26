package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class BasePara_single2plannar_26_0_Test {

    @Test
    public void shouldThrowExceptionForOddNumberOfParameters() {
        BasePara basePara = mock(BasePara.class);
        when(basePara.getQueryparams()).thenReturn(new String[] { "param1", "param2", "param3" });
        assertThrows(OddParamsOfArrayInLoader.class, () -> basePara.single2plannar());
    }

    @Test
    public void shouldReturnCorrectPlanarArrayForEvenNumberOfParameters() {
        BasePara basePara = mock(BasePara.class);
        when(basePara.getQueryparams()).thenReturn(new String[] { "param1", "param2", "param3", "param4" });
        String[][] expected = { { "param1", "param2" }, { "param3", "param4" } };
        assertEquals(expected, basePara.single2plannar());
        verify(basePara).getQueryparams();
    }
}
