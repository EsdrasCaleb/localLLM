package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BasePara_single2plannar_26_0_Test {

    @Test
    @DisplayName("Test single2plannar method")
    public void testSingle2plannar() {
        BasePara basePara = new BasePara();
        // Test when queryparams is null
        basePara.setQueryparams(null);
        assertNull(basePara.single2plannar());
        // Test when queryparams has an odd number of elements
        basePara.setQueryparams(new String[] { "param1", "value1", "param2" });
        assertThrows(OddParamsOfArrayInLoader.class, () -> basePara.single2plannar());
        // Test when queryparams has an even number of elements
        basePara.setQueryparams(new String[] { "param1", "value1", "param2", "value2" });
        String[][] expected = { { "param1", "param2" }, { "value1", "value2" } };
        assertArrayEquals(expected, basePara.single2plannar());
    }
}
