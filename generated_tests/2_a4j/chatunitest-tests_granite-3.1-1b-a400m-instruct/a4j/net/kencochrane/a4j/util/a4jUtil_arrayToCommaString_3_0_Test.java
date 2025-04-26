package net.kencochrane.a4j.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_arrayToCommaString_3_0_Test {

    @Test
    void testArrayToString() {
        a4jUtil util = new a4jUtil();
        ArrayList list = new ArrayList<>();
        list.add("apple");
        list.add("banana");
        list.add("cherry");
        String result = util.arrayToCommaString(list);
        assertEquals("apple, banana, cherry", result);
    }

    @Test
    void testArrayToStringNull() {
        a4jUtil util = new a4jUtil();
        ArrayList list = new ArrayList<>();
        list.add(null);
        String result = util.arrayToCommaString(list);
        assertEquals("null, ", result);
    }
}
