package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class FullProduct_addSimilarItem_7_0_Test {

    private FullProduct fullProduct;

    private MiniProduct mockMiniProduct;

    @BeforeEach
    public void setUp() {
        fullProduct = new FullProduct();
        mockMiniProduct = Mockito.mock(MiniProduct.class);
    }

    @Test
    public void testAddSimilarItem() throws Exception {
        // Initial state check
        Field similarItemsField = FullProduct.class.getDeclaredField("similarItems");
        similarItemsField.setAccessible(true);
        ArrayList<MiniProduct> similarItems = (ArrayList<MiniProduct>) similarItemsField.get(fullProduct);
        assertTrue(similarItems.isEmpty());
        // Add a similar item
        fullProduct.addSimilarItem(mockMiniProduct);
        // Verify that the item was added
        similarItems = (ArrayList<MiniProduct>) similarItemsField.get(fullProduct);
        assertEquals(1, similarItems.size());
        assertSame(mockMiniProduct, similarItems.get(0));
    }
}
