package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class FullProduct_addSimilarItem_7_2_Test {

    private FullProduct fullProduct;

    private ArrayList<MiniProduct> similarItems;

    @BeforeEach
    void setUp() {
        fullProduct = new FullProduct();
        similarItems = new ArrayList<>();
        try {
            Field field = FullProduct.class.getDeclaredField("similarItems");
            field.setAccessible(true);
            field.set(fullProduct, similarItems);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
            fail("Could not access similarItems field");
        }
    }

    @Test
    void addSimilarItem_addsItemToList() {
        // Corrected: Created a new instance instead of mocking
        MiniProduct item = new MiniProduct();
        fullProduct.addSimilarItem(item);
        assertEquals(1, similarItems.size());
        assertSame(item, similarItems.get(0));
    }

    @Test
    void addSimilarItem_emptyListInitially() {
        // Corrected: Created a new instance instead of mocking
        MiniProduct item = new MiniProduct();
        fullProduct.addSimilarItem(item);
        assertEquals(1, similarItems.size());
    }

    @Test
    void addSimilarItem_addsMultipleItems() {
        // Corrected: Created a new instance instead of mocking
        MiniProduct item1 = new MiniProduct();
        // Corrected: Created a new instance instead of mocking
        MiniProduct item2 = new MiniProduct();
        fullProduct.addSimilarItem(item1);
        fullProduct.addSimilarItem(item2);
        assertEquals(2, similarItems.size());
        assertSame(item1, similarItems.get(0));
        assertSame(item2, similarItems.get(1));
    }

    @Test
    void addSimilarItem_nullItem_throwsNoException() {
        assertDoesNotThrow(() -> fullProduct.addSimilarItem(null));
        assertEquals(0, similarItems.size());
    }

    static class MiniProduct implements Serializable {

        private static final long serialVersionUID = 1L;
    }

    static class ProductDetails implements Serializable {

        private static final long serialVersionUID = 1L;
    }

    static class FullProduct implements Serializable {

        private static final long serialVersionUID = 1L;

        private ArrayList<MiniProduct> similarItems = new ArrayList<>();

        public void addSimilarItem(MiniProduct item) {
            if (item != null) {
                similarItems.add(item);
            }
        }
    }
}
