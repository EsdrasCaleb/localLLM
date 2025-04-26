package net.kencochrane.a4j.beans;

import java.util.logging.Logger;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;

@MockitoSettings
@ExtendWith(MockitoExtension.class)
public class Item_toString_18_0_Test {

    @Mock
    private a4jUtil jawsUtil;

    @InjectMocks
    private Item item;

    private static final Logger logger = Logger.getLogger(Item.class.getName());

    @BeforeEach
    public void setup() {
        item = new Item();
    }

    @Test
    public void testToString() {
        item.setAsin("1234567890");
        item.setProductName("Test Product");
        item.setQuantity("10");
        String expected = "Asin = 1234567890\nName = Test Product\nquantity = 10";
        assertEquals(expected, item.toString());
    }

    @Test
    public void testGetCleanPrice() {
        item.setAsin("1234567890");
        item.setProductName("Test Product");
        item.setQuantity("10");
        item.setOurPrice("100.00");
        String expected = "100.00";
        assertEquals(expected, item.getCleanPrice());
    }

    @Test
    public void testGetCleanPriceNull() {
        item.setAsin(null);
        item.setProductName(null);
        item.setQuantity(null);
        item.setOurPrice(null);
        assertThrows(NullPointerException.class, () -> item.getCleanPrice());
    }
}

class Item implements Serializable {

    private String asin;

    private String productName;

    private String quantity;

    private String ourPrice;

    public void setAsin(String asin) {
        this.asin = asin;
    }

    public void setProductName(String productName) {
        this.productName = productName;
    }

    public void setQuantity(String quantity) {
        this.quantity = quantity;
    }

    public void setOurPrice(String ourPrice) {
        this.ourPrice = ourPrice;
    }

    @Override
    public String toString() {
        return "Asin = " + asin + "\nName = " + productName + "\nquantity = " + quantity;
    }

    public String getCleanPrice() {
        return ourPrice == null ? "0.00" : ourPrice;
    }
}
