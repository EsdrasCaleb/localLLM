package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BasePara_single2plannar_26_0_Test {

    private BasePara basePara;

    @BeforeEach
    public void setUp() {
        basePara = new BasePara();
    }

    @Test
    public void testSingle2plannar_nullInput() {
        basePara.setQueryparams(null);
        String[][] result = basePara.single2plannar();
        assertNull(result);
    }

    @Test
    public void testSingle2plannar_oddLengthInput() {
        basePara.setQueryparams(new String[] { "key1", "value1", "key2" });
        String[][] result = basePara.single2plannar();
        assertNull(result);
    }

    @Test
    public void testSingle2plannar_evenLengthInput() {
        basePara.setQueryparams(new String[] { "key1", "value1", "key2", "value2" });
        String[][] result = basePara.single2plannar();
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        assertEquals(2, result[1].length);
        assertEquals("key1", result[0][0]);
        assertEquals("value1", result[1][0]);
        assertEquals("key2", result[0][1]);
        assertEquals("value2", result[1][1]);
    }
}
