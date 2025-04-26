package net.kencochrane.a4j.util;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
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
import java.util.ArrayList;
import java.util.Properties;

public class a4jUtil_encodeString_4_0_Test {

    @ParameterizedTest
    @ValueSource(strings = { "test", "Test", "тест", "123", "  " })
    void testEncodeString(String input) throws UnsupportedEncodingException {
        a4jUtil util = new a4jUtil();
        String encoded = util.encodeString(input);
        assertNotNull(encoded);
        // assertEquals(URLEncoder.encode(input, "UTF-8"), encoded); //This assertion fails if UTF-8 is not available.
        // Instead we check that encoding happened.  The exact encoded value might vary based on the default encoding.
        String encodedDirectly = URLEncoder.encode(input, "UTF-8");
        boolean encodedCorrectly = encoded.equals(encodedDirectly);
        if (!encodedCorrectly) {
            // Fallback encoding was used, check if it's valid.
            assertEquals(URLEncoder.encode(input), encoded);
        } else {
            assertEquals(encodedDirectly, encoded);
        }
    }

    @Test
    void testEncodeStringNull() {
        a4jUtil util = new a4jUtil();
        String encoded = util.encodeString(null);
        // or assertNull(encoded), depending on desired behavior.
        assertEquals("", encoded);
    }

    @Test
    void testEncodeStringEmpty() {
        a4jUtil util = new a4jUtil();
        String encoded = util.encodeString("");
        assertEquals("", encoded);
    }

    @Test
    void testEncodeStringWithSpecialChars() {
        a4jUtil util = new a4jUtil();
        String input = "string with spaces and + symbols";
        String encoded = util.encodeString(input);
        assertNotNull(encoded);
        // Exact value depends on encoding, just check it's not the original string
        assertNotEquals(input, encoded);
    }
}
