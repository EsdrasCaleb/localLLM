package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductInfo_toString_9_3_Test {

    @Test
    void testToString_withProducts() throws NoSuchFieldException, IllegalAccessException {
        ProductDetails product1 = new ProductDetails();
        product1.setName("Product 1");
        product1.setPrice("10.99");
        ProductDetails product2 = new ProductDetails();
        product2.setName("Product 2");
        product2.setPrice("20.50");
        ProductInfo productInfo = new ProductInfo();
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, new ArrayList<>(Arrays.asList(product1, product2)));
        productInfo.setTotalResults("2");
        productInfo.setTotalPages("1");
        String expectedOutput = "Total results = 2\n" + "Total pages = 1\n" + "Name: Product 1, Price: 10.99\n" + "Name: Product 2, Price: 20.50\n" + "# of products = 2\n";
        assertEquals(expectedOutput, productInfo.toString());
    }

    @Test
    void testToString_withoutProducts() throws NoSuchFieldException, IllegalAccessException {
        ProductInfo productInfo = new ProductInfo();
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, null);
        productInfo.setTotalResults("0");
        productInfo.setTotalPages("0");
        String expectedOutput = "Total results = 0\n" + "Total pages = 0\n" + "products is null \n";
        assertEquals(expectedOutput, productInfo.toString());
    }

    @Test
    void testToString_emptyProducts() throws NoSuchFieldException, IllegalAccessException {
        ProductInfo productInfo = new ProductInfo();
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, new ArrayList<>());
        productInfo.setTotalResults("0");
        productInfo.setTotalPages("0");
        String expectedOutput = "Total results = 0\n" + "Total pages = 0\n" + "# of products = 0\n";
        assertEquals(expectedOutput, productInfo.toString());
    }

    class ProductDetails {

        private String name;

        private String price;

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getPrice() {
            return price;
        }

        public void setPrice(String price) {
            this.price = price;
        }

        @Override
        public String toString() {
            return "Name: " + name + ", Price: " + price;
        }
    }
}
