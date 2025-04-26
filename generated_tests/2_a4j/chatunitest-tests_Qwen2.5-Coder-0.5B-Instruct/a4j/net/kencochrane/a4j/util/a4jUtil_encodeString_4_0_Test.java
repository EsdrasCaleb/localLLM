// Test class
package net.kencochrane.a4j.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class a4jUtil_encodeString_4_0_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestClass {

        @Mock
        private a4jUtil a4jUtil;

        @Test
        public void testEncodeString() {
            String searchTerm = "Hello, World!";
            String expectedEncoded = "Hello%2C%20World%21!";
            // <Buggy Line>: non-static method encodeString(java.lang.String) cannot be referenced from a static context
            when(a4jUtil.encodeString(searchTerm)).thenReturn(expectedEncoded);
            String result = a4jUtil.encodeString(searchTerm);
            assertEquals(expectedEncoded, result);
        }
    }
}
