package net.kencochrane.a4j.util;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
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

public class a4jUtil_stripString_1_2_Test {

    @Test
    public void testStripString() {
        // Arrange
        String allowedChars = "abc";
        String inputString = "abcabcabc";
        // Act
        String expectedOutput = "abc";
        // Assert
        assertEquals(expectedOutput, stripString(inputString, allowedChars));
    }

    // Method implementation
    public String stripString(String allowedChars, String inputString) {
        StringBuilder returnString = new StringBuilder();
        String validString = allowedChars;
        char checkChar;
        for (int x = 0; x < inputString.length(); x++) {
            checkChar = inputString.charAt(x);
            if (validString.indexOf(checkChar) != -1) {
                returnString.append(checkChar);
            }
        }
        return returnString.toString();
    }
}
