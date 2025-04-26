package net.kencochrane.a4j.data;

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

public class Query_AddToExistingCart_9_0_Test {

    @Test
    public void testAddToExistingCart() {
        // Arrange
        Query query = new Query();
        query.serverURL = "http://test.com";
        query.associatesID = "testID";
        query.token = "testToken";
        query.searchType = "testType";
        query.type = "testType";
        query.page = "testPage";
        query.offer = "testOffer";
        query.searchValues = new ArrayList<>();
        String ASIN = "testASIN";
        String quantity = "1";
        String cartId = "testCartId";
        String hmac = "testHmac";
        // Act
        String result = query.AddToExistingCart(ASIN, quantity, cartId, hmac);
        // Assert
        assertEquals("http://test.com?ShoppingCart=add&f=xml&dev-t=testToken&t=testID&Asin." + ASIN + "=1&CartId=testCartId&Hmac=testHmac", result);
    }
}
