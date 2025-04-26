package net.kencochrane.a4j;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

public class A4j_KeywordSearch_2_3_Test {

    @Test
    public void keywordSearchTest() {
        // Arrange
        String searchTerm = "test";
        String productLine = "testLine";
        String type = "testType";
        String page = "1";
        A4j a4j = new A4j();
        Search search = Mockito.mock(Search.class);
        // Act
        when(search.Keyword(searchTerm, productLine, type, page)).thenReturn(new ProductInfo());
        ProductInfo result = a4j.KeywordSearch(searchTerm, productLine, type, page);
        // Assert
        assertEquals(new ProductInfo(), result);
    }
}
