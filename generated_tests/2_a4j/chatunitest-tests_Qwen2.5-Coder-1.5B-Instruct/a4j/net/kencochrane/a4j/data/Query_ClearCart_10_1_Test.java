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

public class Query_ClearCart_10_1_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Query query;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testClearCart() throws Exception {
        // Arrange
        String cartId = "12345";
        // example HMAC value
        String hmac = "abcdefg";
        // Act
        String result = query.ClearCart(cartId, hmac);
        // Assert
        assertEquals("http://xml.amazon.com/onca/xml3?ShoppingCart=clear&f=xml&dev-t=[[developer's token goes here]]&t=[associates ID goes here]&CartId=[12345]&Hmac=[abcdefg]", result);
    }
}
