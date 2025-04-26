package net.kencochrane.a4j.beans;

import net.kencochrane.a4j.beans.FullProduct;
import net.kencochrane.a4j.beans.MiniProduct;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_addAccessory_6_0_Test {

    private FullProduct fullProduct;

    @BeforeEach
    void setUp() {
        fullProduct = new FullProduct();
    }

    @Test
    void testAddAccessory_nullProduct() {
        fullProduct.addAccessory(null);
        assertEquals(0, fullProduct.getAccessories().size());
    }

    @Test
    void testAddAccessory_validProduct() {
        MiniProduct miniProduct = new MiniProduct();
        fullProduct.addAccessory(miniProduct);
        assertEquals(1, fullProduct.getAccessories().size());
        assertSame(miniProduct, fullProduct.getAccessories().get(0));
    }

    @Test
    void testAddAccessory_multipleProducts() {
        MiniProduct miniProduct1 = new MiniProduct();
        MiniProduct miniProduct2 = new MiniProduct();
        fullProduct.addAccessory(miniProduct1);
        fullProduct.addAccessory(miniProduct2);
        assertEquals(2, fullProduct.getAccessories().size());
        assertSame(miniProduct1, fullProduct.getAccessories().get(0));
        assertSame(miniProduct2, fullProduct.getAccessories().get(1));
    }
}
