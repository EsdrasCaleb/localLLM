package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class BasePara_single2plannar_26_0_Test {

    private BasePara basePara;

    @BeforeEach
    void setUp() {
        basePara = new BasePara();
    }

    @Test
    void testSingle2plannar_NullQueryParams() {
        basePara.setQueryparams(null);
        String[][] result = basePara.single2plannar();
        assertNull(result, "Expected null when queryparams is null");
    }

    @Test
    void testSingle2plannar_OddLengthQueryParams() {
        basePara.setQueryparams(new String[] { "key1", "value1", "key2" });
        Exception exception = assertThrows(OddParamsOfArrayInLoader.class, () -> {
            basePara.single2plannar();
        });
        assertEquals("Loader加载数据时，所传进来的参数为奇数个！", exception.getMessage());
    }

    @Test
    void testSingle2plannar_EvenLengthQueryParams() {
        basePara.setQueryparams(new String[] { "key1", "value1", "key2", "value2" });
        String[][] result = basePara.single2plannar();
        assertNotNull(result, "Expected non-null result for even length queryparams");
        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        assertEquals("key1", result[0][0]);
        assertEquals("value1", result[1][0]);
        assertEquals("key2", result[0][1]);
        assertEquals("value2", result[1][1]);
    }
}
