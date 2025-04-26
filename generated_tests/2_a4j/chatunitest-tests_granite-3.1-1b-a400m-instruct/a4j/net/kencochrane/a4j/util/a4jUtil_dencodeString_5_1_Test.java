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

class a4jUtil_dencodeString_5_1_Test {

    @Test
    void testEncodeString() {
        a4jUtil util = new a4jUtil();
        String originalString = "Hello, World!";
        String encodedString = util.dencodeString(originalString);
        assertEquals(originalString + "!", encodedString);
        originalString = "Hello, World!";
        encodedString = util.dencodeString(originalString);
        assertEquals(originalString + "!", encodedString);
        originalString = "Hello, World!";
        util.dencodeString(originalString);
        assertEquals(originalString + "!", util.dencodeString(originalString));
    }
}
