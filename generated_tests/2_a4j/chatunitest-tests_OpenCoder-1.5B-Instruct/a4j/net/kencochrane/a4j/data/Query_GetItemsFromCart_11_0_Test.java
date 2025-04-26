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
        // Arrange
        Query query = Mockito.mock(Query.class);
        String cartId = "12345";
        String hmac = "abcdefg";
        String expectedUrl = "http://xml.amazon.com/onca/xml3?ShoppingCart=get&f=xml&dev-t=[ [developer's token goes here]" + "&t=[associates ID goes here]" + "&CartId=[cart ID goes here]" + "&Hmac=[HMAC goes here]";
        // Act
        String resultUrl = query.GetItemsFromCart(cartId, hmac);
        // Assert
        assertEquals(expectedUrl, resultUrl);
    }
}
