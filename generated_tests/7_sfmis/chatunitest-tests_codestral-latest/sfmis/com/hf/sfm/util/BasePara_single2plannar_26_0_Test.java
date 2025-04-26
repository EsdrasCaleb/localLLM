package com.hf.sfm.util;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class BasePara_single2plannar_26_0_Test {

    @InjectMocks
    private BasePara basePara;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testSingle2plannarNullQueryParams() throws Exception {
        setQueryParams(null);
        assertNull(basePara.single2plannar());
    }

    @Test
    void testSingle2plannarOddLengthQueryParams() throws Exception {
        setQueryParams(new String[] { "param1", "value1", "param2" });
        assertNull(basePara.single2plannar());
    }

    @Test
    void testSingle2plannarEvenLengthQueryParams() throws Exception {
        setQueryParams(new String[] { "param1", "value1", "param2", "value2" });
        String[][] result = basePara.single2plannar();
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        assertEquals("param1", result[0][0]);
        assertEquals("value1", result[1][0]);
        assertEquals("param2", result[0][1]);
        assertEquals("value2", result[1][1]);
    }

    private void setQueryParams(String[] queryParams) throws Exception {
        Field field = BasePara.class.getDeclaredField("queryparams");
        field.setAccessible(true);
        field.set(basePara, queryParams);
    }
}
