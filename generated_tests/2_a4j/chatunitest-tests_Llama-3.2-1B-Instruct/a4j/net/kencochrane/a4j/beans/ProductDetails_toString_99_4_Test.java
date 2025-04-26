package net.kencochrane.a4j.beans;

import java.math.BigDecimal;
import java.util.List;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.text.DecimalFormat;

public class ProductDetails_toString_99_4_Test {

    @Test
    public void testGetAsin() {
        ProductDetails product = new ProductDetails();
        assertNotNull(product.getAsin());
        assertEquals("", product.getAsin());
        product.setAsin("1234567890");
        assertNotNull(product.getAsin());
        assertEquals("1234567890", product.getAsin());
    }

    @Test
    public void testSetAsin() {
        ProductDetails product = new ProductDetails();
        product.setAsin("1234567890");
        assertNotNull(product.getAsin());
        assertEquals("1234567890", product.getAsin());
        product.setAsin(null);
        assertNull(product.getAsin());
    }

    @Test
    public void testGetCatalog() {
        ProductDetails product = new ProductDetails();
        assertNotNull(product.getCatalog());
        assertEquals("", product.getCatalog());
        product.setCatalog("catalog");
        assertNotNull(product.getCatalog());
        assertEquals("catalog", product.getCatalog());
    }

    @Test
    public void testSetCatalog() {
        ProductDetails product = new ProductDetails();
        product.setCatalog("catalog");
        assertNotNull(product.getCatalog());
        assertEquals("catalog", product.getCatalog());
        product.setCatalog(null);
        assertNull(product.getCatalog());
    }

    @Test
    public void testGetReleaseDate() {
        ProductDetails product = new ProductDetails();
        assertNotNull(product.getReleaseDate());
        assertEquals("", product.getReleaseDate());
        product.setReleaseDate("2022-01-01");
        assertNotNull(product.getReleaseDate());
        assertEquals("2022-01-01", product.getReleaseDate());
    }

    @Test
    public void testSetReleaseDate() {
        ProductDetails product = new ProductDetails();
        product.setReleaseDate("2022-01-01");
        assertNotNull(product.getReleaseDate());
        assertEquals("2022-01-01", product.getReleaseDate());
        product.setReleaseDate(null);
        assertNull(product.getReleaseDate());
    }

    @Test
    public void testGetManufacturer() {
        ProductDetails product = new ProductDetails();
        assertNotNull(product.getManufacturer());
        assertEquals("", product.getManufacturer());
        product.setManufacturer("Manufacturer");
        assertNotNull(product.getManufacturer());
        assertEquals("Manufacturer", product.getManufacturer());
    }

    @Test
    public void testSetManufacturer() {
        ProductDetails product = new ProductDetails();
        product.setManufacturer("Manufacturer");
        assertNotNull(product.getManufacturer());
        assertEquals("Manufacturer", product.getManufacturer());
        product.setManufacturer(null);
        assertNull(product.getManufacturer());
    }

    @Test
    public void testGetImageURLSmall() {
        ProductDetails product = new ProductDetails();
        assertNotNull(product.getImageUrlSmall());
        assertEquals("", product.getImageUrlSmall());
        product.setImageUrlSmall("url");
        assertNotNull(product.getImageUrlSmall());
        assertEquals("url", product.getImageUrlSmall());
    }

    @Test
    public void testSetImageURLSmall() {
        ProductDetails product = new ProductDetails();
        product.setImageUrlSmall("url");
        assertNotNull(product.getImageUrlSmall());
        assertEquals("url", product.getImageUrlSmall());
        product.setImageUrlSmall(null);
        assertNull(product.getImageUrlSmall());
    }

    @Test
    public void testGetImageURLMedium() {
        ProductDetails product = new ProductDetails();
        assertNotNull(product.getImageUrlMedium());
        assertEquals("", product.getImageUrlMedium());
        product.setImageUrlMedium("url");
        assertNotNull(product.getImageUrlMedium());
        assertEquals("url", product.getImageUrlMedium());
    }

    @Test
    public void testSetImageURLMedium() {
        ProductDetails product = new ProductDetails();
        product.setImageUrlMedium("url");
        assertNotNull(product.getImageUrlMedium());
        assertEquals("url", product.getImageUrlMedium());
        product.setImageUrlMedium(null);
        assertNull(product.getImageUrlMedium());
    }

    @Test
    public void testGetImageURLLarge() {
        ProductDetails product = new ProductDetails();
        assertNotNull(product.getImageUrlLarge());
        assertEquals("", product.getImageUrlLarge());
        product.setImageUrlLarge("url");
        assertNotNull(product.getImageUrlLarge());
        assertEquals("url", product.getImageUrlLarge());
    }

    @Test
    public void testSetImageURLLarge() {
        ProductDetails product = new ProductDetails();
        product.setImageUrlLarge("url");
        assertNotNull(product.getImageUrlLarge());
        assertEquals("url", product.getImageUrlLarge());
        product.setImageUrlLarge(null);
        assertNull(product.getImageUrlLarge());
    }

    @Test
    public void testGetMedia() {
        ProductDetails product = new ProductDetails();
        assertNotNull(product.getMedia());
        assertEquals("", product.getMedia());
    }
}
