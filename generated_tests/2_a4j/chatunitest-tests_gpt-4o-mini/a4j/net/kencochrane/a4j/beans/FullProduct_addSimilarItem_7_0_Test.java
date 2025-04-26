package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class FullProduct_addSimilarItem_7_0_Test {

    private FullProduct fullProduct;

    @BeforeEach
    void setUp() {
        fullProduct = new FullProduct();
    }

    @Test
    void testAddSimilarItem() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        MiniProduct miniProduct = new MiniProduct();
        // Act
        fullProduct.addSimilarItem(miniProduct);
        // Assert
        Field similarItemsField = FullProduct.class.getDeclaredField("similarItems");
        similarItemsField.setAccessible(true);
        ArrayList<MiniProduct> similarItems = (ArrayList<MiniProduct>) similarItemsField.get(fullProduct);
        assertEquals(1, similarItems.size());
        assertEquals(miniProduct, similarItems.get(0));
    }

    @Test
    void testAddMultipleSimilarItems() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        MiniProduct miniProduct1 = new MiniProduct();
        MiniProduct miniProduct2 = new MiniProduct();
        // Act
        fullProduct.addSimilarItem(miniProduct1);
        fullProduct.addSimilarItem(miniProduct2);
        // Assert
        Field similarItemsField = FullProduct.class.getDeclaredField("similarItems");
        similarItemsField.setAccessible(true);
        ArrayList<MiniProduct> similarItems = (ArrayList<MiniProduct>) similarItemsField.get(fullProduct);
        assertEquals(2, similarItems.size());
        assertEquals(miniProduct1, similarItems.get(0));
        assertEquals(miniProduct2, similarItems.get(1));
    }
}
