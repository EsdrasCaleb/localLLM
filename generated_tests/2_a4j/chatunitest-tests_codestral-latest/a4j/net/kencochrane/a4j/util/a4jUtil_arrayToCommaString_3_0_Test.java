package net.kencochrane.a4j.util;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class a4jUtil_arrayToCommaString_3_0_Test {

    @InjectMocks
    private a4jUtil a4jUtil;

    @BeforeEach
    public void setUp() {
        a4jUtil = new a4jUtil();
    }

    @Test
    public void testArrayToCommaString_NullList() {
        assertEquals("", a4jUtil.arrayToCommaString(null));
    }

    @Test
    public void testArrayToCommaString_EmptyList() {
        assertEquals("", a4jUtil.arrayToCommaString(new ArrayList<>()));
    }

    @Test
    public void testArrayToCommaString_SingleElement() {
        ArrayList<String> list = new ArrayList<>();
        list.add("test");
        assertEquals("test", a4jUtil.arrayToCommaString(list));
    }

    @Test
    public void testArrayToCommaString_MultipleElements() {
        ArrayList<String> list = new ArrayList<>();
        list.add("test1");
        list.add("test2");
        list.add("test3");
        assertEquals("test1, test2, test3", a4jUtil.arrayToCommaString(list));
    }

    @Test
    public void testArrayToCommaString_WithNullElements() {
        ArrayList<String> list = new ArrayList<>();
        list.add("test1");
        list.add(null);
        list.add("test3");
        assertEquals("test1, test3", a4jUtil.arrayToCommaString(list));
    }

    @Test
    public void testArrayToCommaString_AllNullElements() {
        ArrayList<String> list = new ArrayList<>();
        list.add(null);
        list.add(null);
        assertEquals("", a4jUtil.arrayToCommaString(list));
    }
}
