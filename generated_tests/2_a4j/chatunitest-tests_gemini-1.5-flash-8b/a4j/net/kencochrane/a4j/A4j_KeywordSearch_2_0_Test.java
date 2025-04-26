package net.kencochrane.a4j;

import net.kencochrane.a4j.A4j;
import net.kencochrane.a4j.DAO.Search;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.beans.*;

class A4j_KeywordSearch_2_0_Test {

    @Test
    void keywordSearch_validInput_returnsProductInfo() {
        Search searchMock = Mockito.mock(Search.class);
        ProductInfo expectedProductInfo = new ProductInfo();
        Mockito.when(searchMock.Keyword("testTerm", "testLine", "testType", "1")).thenReturn(expectedProductInfo);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.KeywordSearch("testTerm", "testLine", "testType", "1");
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void keywordSearch_nullSearchTerm_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.KeywordSearch(null, "testLine", "testType", "1");
        assertNull(actualProductInfo);
    }

    @Test
    void keywordSearch_emptySearchTerm_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.KeywordSearch("", "testLine", "testType", "1");
        assertNull(actualProductInfo);
    }

    @Test
    void keywordSearch_nullProductLine_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.KeywordSearch("testTerm", null, "testType", "1");
        assertNull(actualProductInfo);
    }

    @Test
    void keywordSearch_emptyProductLine_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.KeywordSearch("testTerm", "", "testType", "1");
        assertNull(actualProductInfo);
    }

    @Test
    void keywordSearch_searchReturnsNull_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        Mockito.when(searchMock.Keyword("testTerm", "testLine", "testType", "1")).thenReturn(null);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.KeywordSearch("testTerm", "testLine", "testType", "1");
        assertNull(actualProductInfo);
    }

    @Test
    void keywordSearch_nullProductType_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.KeywordSearch("testTerm", "testLine", null, "1");
        assertNull(actualProductInfo);
    }

    @Test
    void keywordSearch_emptyProductType_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.KeywordSearch("testTerm", "testLine", "", "1");
        assertNull(actualProductInfo);
    }
}
