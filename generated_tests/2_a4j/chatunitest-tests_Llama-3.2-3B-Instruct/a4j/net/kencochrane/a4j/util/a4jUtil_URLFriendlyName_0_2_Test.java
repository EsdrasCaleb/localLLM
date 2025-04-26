package net.kencochrane.a4j.util;

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
import java.util.ArrayList;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class a4jUtil_URLFriendlyName_0_2_Test {

    @Mock
    private LoadProperties instance;

    @InjectMocks
    private a4jUtil a4jUtil;

    @Test
    public void testURLFriendlyName_EmptyString_ReturnsEmptyString() {
        when(instance.getProperties()).thenReturn(new Properties());
        String result = a4jUtil.URLFriendlyName("");
        assertEquals("", result);
    }

    @Test
    public void testURLFriendlyName_NullString_ThrowsNullPointerException() {
        when(instance.getProperties()).thenReturn(new Properties());
        assertThrows(NullPointerException.class, () -> a4jUtil.URLFriendlyName(null));
    }

    @Test
    public void testURLFriendlyName_SingleCharacter_ReturnsSingleCharacter() {
        when(instance.getProperties()).thenReturn(new Properties());
        String result = a4jUtil.URLFriendlyName("a");
        assertEquals("a", result);
    }

    @Test
    public void testURLFriendlyName_Space_ReturnsURLSeparator() {
        when(instance.getProperties()).thenReturn(new Properties());
        String result = a4jUtil.URLFriendlyName("test");
        assertEquals("test", a4jUtil.URLFriendlyName("test"));
    }

    @Test
    public void testURLFriendlyName_MultipleSpaces_ReturnsURLSeparator() {
        when(instance.getProperties()).thenReturn(new Properties());
        String result = a4jUtil.URLFriendlyName("test space test");
        assertEquals("testspace-test", a4jUtil.URLFriendlyName("test space test"));
    }

    @Test
    public void testURLFriendlyName_InvalidCharacter_ReturnsEmptyString() {
        when(instance.getProperties()).thenReturn(new Properties());
        String result = a4jUtil.URLFriendlyName("test!@#$%^&*()_+");
        assertEquals("", result);
    }

    @Test
    public void testURLFriendlyName_InvalidCharacterWithURLSeparator_ReturnsEmptyString() {
        when(instance.getProperties()).thenReturn(new Properties());
        String result = a4jUtil.URLFriendlyName("test!@#$%^&*()_+");
        assertEquals("", result);
    }

    @Test
    public void testURLFriendlyName_InvalidCharacterWithURLSeparatorAndSpace_ReturnsEmptyString() {
        when(instance.getProperties()).thenReturn(new Properties());
        String result = a4jUtil.URLFriendlyName("test!@#$%^&*()_+ ");
        assertEquals("", result);
    }
}
