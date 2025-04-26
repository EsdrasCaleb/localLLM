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

public class a4jUtil_stripString_1_4_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestUtil {

        @Mock
        private a4jUtil a4jUtil;

        @BeforeEach
        void setUp() {
            MockitoAnnotations.openMocks(this);
        }

        @Test
        public void testStripString() {
            String allowedChars = "abc";
            String string = "xyz";
            // <Buggy Line>: non-static method stripString(java.lang.String,java.lang.String) cannot be referenced from a static context
            String result = a4jUtil.stripString(allowedChars, string);
            assertEquals("x", result);
        }
    }
}
