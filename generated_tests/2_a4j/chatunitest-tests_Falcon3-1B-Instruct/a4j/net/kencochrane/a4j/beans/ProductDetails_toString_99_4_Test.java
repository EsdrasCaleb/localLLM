package net.kencochrane.a4j.beans;

import static org.junit.Assert.*;
import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.math.BigDecimal;
import java.text.DecimalFormat;

public class ProductDetails_toString_99_4_Test {

    @Test
    public void testGetAsin() {
        ProductDetails product = new ProductDetails();
        assertEquals("asin", product.getAsin(), 0);
    }

    @Test
    public void testGetProductName() {
        ProductDetails product = new ProductDetails();
        assertEquals("productName", product.getProductName(), 0);
    }

    @Test
    public void testGetCatalog() {
        ProductDetails product = new ProductDetails();
        assertEquals("catalog", product.getCatalog(), 0);
    }

    @Test
    public void testGetReleaseDate() {
        ProductDetails product = new ProductDetails();
        assertEquals("releaseDate", product.getReleaseDate(), 0);
    }

    @Test
    public void testGetManufacturer() {
        ProductDetails product = new ProductDetails();
        assertEquals("manufacturer", product.getManufacturer(), 0);
    }

    @Test
    public void testGetImageUrlSmall() {
        ProductDetails product = new ProductDetails();
        assertEquals("imageUrlSmall", product.getImageUrlSmall(), 0);
    }

    @Test
    public void testGetImageUrlMedium() {
        ProductDetails product = new ProductDetails();
        assertEquals("imageUrlMedium", product.getImageUrlMedium(), 0);
    }

    @Test
    public void testGetImageUrlLarge() {
        ProductDetails product = new ProductDetails();
        assertEquals("imageUrlLarge", product.getImageUrlLarge(), 0);
    }

    @Test
    public void testGetMedia() {
        ProductDetails product = new ProductDetails();
        assertEquals("media", product.getMedia(), 0);
    }

    @Test
    public void testGetIsbn() {
        ProductDetails product = new ProductDetails();
        assertEquals("isbn", product.getIsbn(), 0);
    }

    @Test
    public void testGetAvailability() {
        ProductDetails product = new ProductDetails();
        assertEquals("availability", product.getAvailability(), 0);
    }

    @Test
    public void testGetMpn() {
        ProductDetails product = new ProductDetails();
        assertEquals("mpn", product.getMpn(), 0);
    }

    @Test
    public void testGetListPrice() {
        ProductDetails product = new ProductDetails();
        assertEquals("listPrice", product.getListPrice(), 0);
    }

    @Test
    public void testGetOurPrice() {
        ProductDetails product = new ProductDetails();
        assertEquals("ourPrice", product.getOurPrice(), 0);
    }

    @Test
    public void testGetUsedPrice() {
        ProductDetails product = new ProductDetails();
        assertEquals("usedPrice", product.getUsedPrice(), 0);
    }

    @Test
    public void testGetThirdPartyNewPrice() {
        ProductDetails product = new ProductDetails();
        assertEquals("thirdPartyNewPrice", product.getThirdPartyNewPrice(), 0);
    }

    @Test
    public void testGetCollectiblePrice() {
        ProductDetails product = new ProductDetails();
        assertEquals("collectiblePrice", product.getCollectiblePrice(), 0);
    }

    @Test
    public void testGetThirdPartyNewCount() {
        ProductDetails product = new ProductDetails();
        assertEquals("thirdPartyNewCount", product.getThirdPartyNewCount(), 0);
    }

    @Test
    public void testGetUsedCount() {
        ProductDetails product = new ProductDetails();
        assertEquals("usedCount", product.getUsedCount(), 0);
    }

    @Test
    public void testGetCollectibleCount() {
        ProductDetails product = new ProductDetails();
        assertEquals("collectibleCount", product.getCollectibleCount(), 0);
    }

    @Test
    public void testGetRefurbishedPrice() {
        ProductDetails product = new ProductDetails();
        assertEquals("refurbishedPrice", product.getRefurbishedPrice(), 0);
    }
}
