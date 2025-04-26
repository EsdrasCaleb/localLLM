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

public class Query_GetItemsFromCart_11_1_Test {

    @Test
    public void testGetItemsFromCart() {
        Query query = new Query();
        String cartId = "12345";
        String hmac = "abcde";
        String expectedUrl = query.serverURL + "?ShoppingCart=get&f=xml&dev-t=" + query.token + "&t=" + query.associatesID + "&CartId=" + cartId + "&Hmac=" + hmac;
        String actualUrl = query.GetItemsFromCart(cartId, hmac);
        assertEquals(expectedUrl, actualUrl);
    }
}
