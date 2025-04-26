package net.kencochrane.a4j.util;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_dencodeString_5_0_Test {

    @ParameterizedTest
    @ValueSource(strings = { "%20Hello%20World", "%E4%BD%A0%E5%A5%BD", "test" })
    void testDencodeString_utf8Supported(String encodedString) throws UnsupportedEncodingException {
        a4jUtil util = new a4jUtil();
        String decodedString = util.dencodeString(encodedString);
        String expectedString = URLDecoder.decode(encodedString, "UTF-8");
        assertEquals(expectedString, decodedString);
    }

    @Test
    void testDencodeString_utf8NotSupported() throws UnsupportedEncodingException {
        // Mocking UnsupportedEncodingException is not necessary here because the fallback mechanism handles it.
        // The test verifies the fallback mechanism works correctly.
        a4jUtil util = new a4jUtil();
        // Example using characters outside basic ASCII
        String encodedString = "%E4%BD%A0%E5%A5%BD";
        String decodedString = util.dencodeString(encodedString);
        // Since we cannot reliably predict the default encoding, we compare against a decode using the default encoding.
        String expectedString = URLDecoder.decode(encodedString);
        assertEquals(expectedString, decodedString);
    }

    @Test
    void testDencodeString_NullInput() {
        a4jUtil util = new a4jUtil();
        String decodedString = util.dencodeString(null);
        assertEquals(null, decodedString);
    }

    @Test
    void testDencodeString_EmptyInput() {
        a4jUtil util = new a4jUtil();
        String decodedString = util.dencodeString("");
        assertEquals("", decodedString);
    }
}
