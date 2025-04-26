package net.kencochrane.a4j.data;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.util.a4jUtil;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.LoadProperties;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.Properties;

@ExtendWith(MockitoExtension.class)
public class Query_RemoveFromCart_13_2_Test {

    @Mock
    private a4jUtil mockUtil;

    @Test
    void testRemoveFromCart() throws Exception {
        Query query = new Query();
        // Use reflection to set private fields for testing
        Field serverURLField = Query.class.getDeclaredField("serverURL");
        serverURLField.setAccessible(true);
        serverURLField.set(query, "http://xml.amazon.com/onca/xml3");
        Field associatesIDField = Query.class.getDeclaredField("associatesID");
        associatesIDField.setAccessible(true);
        associatesIDField.set(query, "test");
        Field searchValuesField = Query.class.getDeclaredField("searchValues");
        searchValuesField.setAccessible(true);
        searchValuesField.set(query, new ArrayList<>());
        // Inject the mock a4jUtil
        Field jawsUtilField = Query.class.getDeclaredField("jawsUtil");
        jawsUtilField.setAccessible(true);
        jawsUtilField.set(query, mockUtil);
        when(mockUtil.encodeString(anyString())).thenReturn("HMAC=");
        String itemId = "17120277375791359165";
        String cartId = "CART";
        String hmac = "hmac";
        String expectedURL = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&f=xml&dev-t=DSB0XDDW1GQ3S&t=test&Item." + itemId + "&CartId=" + cartId + "&Hmac=HMAC=";
        String actualURL = query.RemoveFromCart(itemId, cartId, hmac);
        assertEquals(expectedURL, actualURL);
        // Test with null values
        assertThrows(NullPointerException.class, () -> query.RemoveFromCart(null, cartId, hmac));
        assertThrows(NullPointerException.class, () -> query.RemoveFromCart(itemId, null, hmac));
        assertThrows(NullPointerException.class, () -> query.RemoveFromCart(itemId, cartId, null));
        // Test with empty values
        String emptyItemId = "";
        String emptyCartId = "";
        String emptyHmac = "";
        String expectedURL2 = "http://xml.amazon.com/onca/xml3?ShoppingCart=remove&f=xml&dev-t=DSB0XDDW1GQ3S&t=test&Item.&CartId=&Hmac=HMAC=";
        String actualURL2 = query.RemoveFromCart(emptyItemId, emptyCartId, emptyHmac);
        assertEquals(expectedURL2, actualURL2);
    }
}
