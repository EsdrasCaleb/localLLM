package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class ThirdPartyProductInfo_toString_3_0_Test {

    @Test
    void testToString_withProducts() {
        ThirdPartyProductDetails product1 = new ThirdPartyProductDetails("Product A", 10.99);
        ThirdPartyProductDetails product2 = new ThirdPartyProductDetails("Product B", 25.50);
        ThirdPartyProductInfo info = new ThirdPartyProductInfo();
        info.setThirdPartyProductDetails(new ThirdPartyProductDetails[] { product1, product2 });
        String expected = "Product A - $10.99\n" + "Product B - $25.50\n" + "# of productOffers = 2";
        assertEquals(expected, info.toString());
    }

    @Test
    void testToString_noProducts() {
        ThirdPartyProductInfo info = new ThirdPartyProductInfo();
        try {
            Field field = ThirdPartyProductInfo.class.getDeclaredField("productOffers");
            field.setAccessible(true);
            field.set(info, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access private field: " + e.getMessage());
        }
        assertEquals("productOffers is null ", info.toString());
    }

    @Test
    void testToString_emptyProducts() {
        ThirdPartyProductInfo info = new ThirdPartyProductInfo();
        info.setThirdPartyProductDetails(new ThirdPartyProductDetails[0]);
        String expected = "# of productOffers = 0";
        assertEquals(expected, info.toString());
    }

    // Inner class for ThirdPartyProductDetails (assuming its structure)
    static class ThirdPartyProductDetails {

        private String productName;

        private double price;

        public ThirdPartyProductDetails() {
        }

        public ThirdPartyProductDetails(String productName, double price) {
            this.productName = productName;
            this.price = price;
        }

        @Override
        public String toString() {
            return productName + " - $" + price;
        }
    }

    static class ThirdPartyProductInfo {

        private ThirdPartyProductDetails[] productOffers;

        public void setThirdPartyProductDetails(ThirdPartyProductDetails[] productOffers) {
            this.productOffers = productOffers;
        }

        @Override
        public String toString() {
            if (productOffers == null) {
                return "productOffers is null ";
            } else if (productOffers.length == 0) {
                return "# of productOffers = 0";
            } else {
                StringBuilder sb = new StringBuilder();
                for (ThirdPartyProductDetails product : productOffers) {
                    sb.append(product.toString()).append("\n");
                }
                sb.append("# of productOffers = ").append(productOffers.length);
                return sb.toString();
            }
        }
    }
}
