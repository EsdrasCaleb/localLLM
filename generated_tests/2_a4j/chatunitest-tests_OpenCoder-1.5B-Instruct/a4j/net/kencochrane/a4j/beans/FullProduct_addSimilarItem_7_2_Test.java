package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_addSimilarItem_7_2_Test {

    public static void main(String[] args) {
        FullProduct product = new FullProduct();
        MiniProduct accessory = new MiniProduct();
        accessory.setName("Glasses");
        product.addSimilarItem(accessory);
        // Output should be 1
        System.out.println(product.similarItems.size());
    }

    @Test
    public void testAddSimilarItem() {
        FullProduct product = new FullProduct();
        MiniProduct accessory = new MiniProduct();
        accessory.setName("Glasses");
        product.addSimilarItem(accessory);
        assertEquals(1, product.similarItems.size());
    }
}
