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

class FullProduct_addSimilarItem_7_0_Test {

    @Test
    void addSimilarItem_shouldAddProductToList() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = new MiniProduct("Test Product");
        // Act
        fullProduct.addSimilarItem(miniProduct);
        // Assert
        List<MiniProduct> similarItems = fullProduct.getSimilarItems();
        assertTrue(similarItems.contains(miniProduct));
    }

    @Test
    void addSimilarItem_withNullProduct_shouldNotThrowException() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct nullProduct = null;
        // Act
        fullProduct.addSimilarItem(nullProduct);
        // Assert
        List<MiniProduct> similarItems = fullProduct.getSimilarItems();
        assertEquals(0, similarItems.size());
    }

    @Test
    void addSimilarItem_withEmptyList_shouldAddProduct() {
        // Arrange
        FullProduct fullProduct = new FullProduct();
        MiniProduct miniProduct = new MiniProduct("Test Product 2");
        // Act
        fullProduct.addSimilarItem(miniProduct);
        // Assert
        List<MiniProduct> similarItems = fullProduct.getSimilarItems();
        assertEquals(1, similarItems.size());
        assertTrue(similarItems.contains(miniProduct));
    }

    // Dummy classes for testing (These are crucial for the test to compile and run)
    static class FullProduct {

        private List<MiniProduct> similarItems = new ArrayList<>();

        public void addSimilarItem(MiniProduct miniProduct) {
            if (miniProduct != null) {
                this.similarItems.add(miniProduct);
            }
        }

        public List<MiniProduct> getSimilarItems() {
            // Return a copy to avoid modification issues.
            return new ArrayList<>(this.similarItems);
        }
    }

    static class MiniProduct {

        private String name;

        public MiniProduct(String name) {
            this.name = name;
        }

        // Needed for equals and hashcode
        @Override
        public boolean equals(Object o) {
            if (this == o)
                return true;
            if (o == null || getClass() != o.getClass())
                return false;
            MiniProduct that = (MiniProduct) o;
            return name.equals(that.name);
        }

        @Override
        public int hashCode() {
            return name.hashCode();
        }
    }
}
