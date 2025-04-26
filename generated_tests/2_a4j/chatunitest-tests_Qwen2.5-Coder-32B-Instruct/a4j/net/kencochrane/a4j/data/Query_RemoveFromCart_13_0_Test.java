package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
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
import java.util.Properties;

public class Query_RemoveFromCart_13_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize protected fields using reflection
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "test");
        Field searchValuesField = Query.class.getDeclaredField("searchValues");
        searchValuesField.setAccessible(true);
        searchValuesField.set(query, new ArrayList<>());
    }

    @Test
    public void testRemoveFromCart() {
        String itemId = "17120277375791359165";
        String cartId = "CART";
        String hmac = "HMAC";
        String encodedHmac = "encodedHMAC";
        when(jawsUtil.encodeString(hmac)).thenReturn(encodedHmac);
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&f=xml&dev-t=DSB0XDDW1GQ3S&t=test&Item." + itemId + "&CartId=" + cartId + "&Hmac=" + encodedHmac;
        String result = query.RemoveFromCart(itemId, cartId, hmac);
        assertEquals(expectedUrl, result);
        verify(jawsUtil, times(1)).encodeString(hmac);
    }
}
