package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

class ShoppingCart_getItem_10_0_Test {

    @Test
    void testGetItem_nullItems() {
        ShoppingCart cart = new ShoppingCart();
        assertNull(cart.getItem("789"));
    }

    // Dummy classes for compilation
    static class Item {

        public String getItemId() {
            return null;
        }

        public String getOurPrice() {
            return null;
        }

        public String getQuantity() {
            return null;
        }
    }

    static class Items {

        public ArrayList getItemsArrayList() {
            return null;
        }
    }

    static class a4jUtil {

        public BigDecimal getPrice(String itemPrice) {
            return new BigDecimal(0.00);
        }
    }
}
