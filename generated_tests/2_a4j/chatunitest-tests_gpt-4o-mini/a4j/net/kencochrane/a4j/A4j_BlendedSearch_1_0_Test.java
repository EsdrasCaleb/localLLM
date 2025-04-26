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

class A4j_BlendedSearch_1_0_Test {

    @InjectMocks
    private A4j a4j;

    @Mock
    private Search searchMock;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testBlendedSearch_ValidInputs() {
        // Arrange
        String searchTerm = "test";
        String type = "type1";
        BlendedSearch expectedResult = new BlendedSearch();
        when(searchMock.Blended(searchTerm, type)).thenReturn(expectedResult);
        // Act
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        // Assert
        assertEquals(expectedResult, result);
        verify(searchMock).Blended(searchTerm, type);
    }

    @Test
    void testBlendedSearch_EmptySearchTerm() {
        // Arrange
        String searchTerm = "";
        String type = "type2";
        BlendedSearch expectedResult = new BlendedSearch();
        when(searchMock.Blended(searchTerm, type)).thenReturn(expectedResult);
        // Act
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        // Assert
        assertEquals(expectedResult, result);
        verify(searchMock).Blended(searchTerm, type);
    }

    @Test
    void testBlendedSearch_NullSearchTerm() {
        // Arrange
        String searchTerm = null;
        String type = "type3";
        BlendedSearch expectedResult = new BlendedSearch();
        when(searchMock.Blended(searchTerm, type)).thenReturn(expectedResult);
        // Act
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        // Assert
        assertEquals(expectedResult, result);
        verify(searchMock).Blended(searchTerm, type);
    }

    @Test
    void testBlendedSearch_NullType() {
        // Arrange
        String searchTerm = "test";
        String type = null;
        BlendedSearch expectedResult = new BlendedSearch();
        when(searchMock.Blended(searchTerm, type)).thenReturn(expectedResult);
        // Act
        BlendedSearch result = a4j.BlendedSearch(searchTerm, type);
        // Assert
        assertEquals(expectedResult, result);
        verify(searchMock).Blended(searchTerm, type);
    }
}
