package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_addAccessory_6_1_Test {

    @Test
    public void testAddAccessory() {
        FullProduct product = new FullProduct();
        MiniProduct miniProduct = new MiniProduct();
        product.addAccessory(miniProduct);
        assertEquals(1, product.accessories.size());
        assertEquals(miniProduct, product.getAccessories().get(0));
    }

    @Test
    public void testAddAccessoryMultiple() {
        FullProduct product = new FullProduct();
        MiniProduct miniProduct1 = new MiniProduct();
        MiniProduct miniProduct2 = new MiniProduct();
        product.addAccessory(miniProduct1);
        product.addAccessory(miniProduct2);
        assertEquals(2, product.accessories.size());
        assertEquals(miniProduct1, product.getAccessories().get(0));
        assertEquals(miniProduct2, product.getAccessories().get(1));
    }
}
