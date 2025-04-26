package net.kencochrane.a4j.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.junit.jupiter.api.extension.ExtendWith;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;
import org.apache.log4j.Logger;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;

@ExtendWith(MockitoExtension.class)
public class a4jUtil_arrayToCommaString_3_1_Test {

    @Mock
    private ArrayList<String> list;

    private a4jUtil a4jUtil;

    @BeforeEach
    public void setup() {
        a4jUtil = Mockito.spy(new a4jUtil());
    }

    @Test
    public void testArrayToCommaString_EmptyList_ReturnsEmptyString() {
        when(list.size()).thenReturn(0);
        when(list.get(0)).thenReturn(null);
        when(list.get(1)).thenReturn(null);
        when(list.get(2)).thenReturn(null);
        when(list.get(3)).thenReturn(null);
        String result = a4jUtil.arrayToCommaString(list);
        assertEquals("", result);
    }

    @Test
    public void testArrayToCommaString_NullList_ReturnsNull() {
        when(list.size()).thenReturn(0);
        when(list.get(0)).thenReturn(null);
        when(list.get(1)).thenReturn(null);
        when(list.get(2)).thenReturn(null);
        when(list.get(3)).thenReturn(null);
        String result = a4jUtil.arrayToCommaString(list);
        assertNull(result);
    }

    @Test
    public void testArrayToCommaString_SingleElement_ReturnsElement() {
        when(list.size()).thenReturn(1);
        when(list.get(0)).thenReturn("Hello");
        when(list.get(1)).thenReturn(null);
        when(list.get(2)).thenReturn(null);
        when(list.get(3)).thenReturn(null);
        String result = a4jUtil.arrayToCommaString(list);
        assertEquals("Hello", result);
    }

    @Test
    public void testArrayToCommaString_MultipleElements_ReturnsCommaSeparatedString() {
        when(list.size()).thenReturn(4);
        when(list.get(0)).thenReturn("Hello");
        when(list.get(1)).thenReturn("World");
        when(list.get(2)).thenReturn("Java");
        when(list.get(3)).thenReturn("Testing");
        String result = a4jUtil.arrayToCommaString(list);
        assertEquals("Hello, World, Java, Testing", result);
    }
}
