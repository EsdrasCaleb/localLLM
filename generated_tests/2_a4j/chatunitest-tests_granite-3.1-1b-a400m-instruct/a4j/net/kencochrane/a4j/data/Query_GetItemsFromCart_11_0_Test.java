package net.kencochrane.a4j.data;

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

public class Query_GetItemsFromCart_11_0_Test {

    @Test
    public void testGetItemsFromCart() {
        // Given
        String serverURL = "http://example.com/api/get_items";
        String associatesID = "12345";
        String hmac = "your_developer_token";
        String cartId = "your_cart_id";
        String token = "your_developer_token";
        String offer = "your_offer";
        // When
        String result = new Query().GetItemsFromCart(cartId, hmac);
        // Then
        assertEquals("http://example.com/api/get_items?ShoppingCart=get&f=xml&dev-t=your_developer_token&t=12345&CartId=your_cart_id&Hmac=your_hmac", result);
    }
}
