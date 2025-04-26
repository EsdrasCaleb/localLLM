package com.hf.sfm.util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class BasePara_single2plannar_26_0_Test {

    private BasePara basePara;

    @BeforeEach
    void setUp() {
        basePara = new BasePara();
    }

    @Test
    void testSingle2plannarNullInput() {
        assertNull(basePara.single2plannar());
    }

    @Test
    void testSingle2plannarOddLengthInput() {
        basePara.setQueryparams(new String[] { "key1" });
        assertNull(basePara.single2plannar());
    }

    @Test
    void testSingle2plannarEvenLengthInput() {
        basePara.setQueryparams(new String[] { "key1", "value1", "key2", "value2" });
        String[][] result = basePara.single2plannar();
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        assertEquals("key1", result[0][0]);
        assertEquals("value1", result[1][0]);
        assertEquals("key2", result[0][1]);
        assertEquals("value2", result[1][1]);
    }

    @Test
    void testSingle2plannarEmptyInput() {
        basePara.setQueryparams(new String[] {});
        assertNull(basePara.single2plannar());
    }

    @Test
    void testSingle2plannarLongInput() {
        String[] params = new String[100];
        for (int i = 0; i < 100; i += 2) {
            params[i] = "key" + i;
            params[i + 1] = "value" + i;
        }
        basePara.setQueryparams(params);
        String[][] result = basePara.single2plannar();
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(50, result[0].length);
        assertEquals("key0", result[0][0]);
        assertEquals("value98", result[1][49]);
    }
}

class OddParamsOfArrayInLoader extends Exception {

    public OddParamsOfArrayInLoader(String msg) {
        super(msg);
    }
}
