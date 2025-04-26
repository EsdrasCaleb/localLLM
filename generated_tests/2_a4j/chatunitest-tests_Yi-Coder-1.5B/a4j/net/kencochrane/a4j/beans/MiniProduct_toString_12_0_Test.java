package net.kencochrane.a4j.beans;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

@RunWith(MockitoJUnitRunner.class)
public class MiniProduct_toString_12_0_Test {

    @Test
    public void testToString() {
        MiniProduct product = new MiniProduct();
        product.setAsin("1234567890");
        product.setName("Test Product");
        product.setManufacturer("Test Manufacturer");
        product.setPrice("$10.00");
        product.setImageURL("https://example.com/image.jpg");
        product.setProductUrl("https://example.com/product.html");
        String expected = "1234567890 \n Test Product \n Test Manufacturer \n $10.00 \n https://example.com/image.jpg";
        String actual = product.toString();
        Assert.assertEquals(expected, actual);
    }
}
