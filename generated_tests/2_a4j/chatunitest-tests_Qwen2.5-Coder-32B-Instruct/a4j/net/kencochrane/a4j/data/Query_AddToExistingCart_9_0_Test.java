package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import net.kencochrane.a4j.util.a4jUtil;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.Properties;

public class Query_AddToExistingCart_9_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Set up the serverURL and associatesID using reflection
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "associatesID123");
    }

    @Test
    public void testAddToExistingCart() {
        // Given
        String ASIN = "B08N5WRWNW";
        String quantity = "2";
        String cartId = "123456789";
        String hmac = "hmacValue";
        // Expected value after encoding
        String expectedHmac = "encodedHMAC";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=add&f=xml&dev-t=DSB0XDDW1GQ3S&t=associatesID123&Asin.B08N5WRWNW=2&CartId=123456789&Hmac=encodedHMAC";
        when(jawsUtil.encodeString(hmac)).thenReturn(expectedHmac);
        // When
        String result = query.AddToExistingCart(ASIN, quantity, cartId, hmac);
        // Then
        assertEquals(expectedUrl, result);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
