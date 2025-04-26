package net.kencochrane.a4j;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
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

public class A4j_DirectorSearch_6_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @ParameterizedTest
    @CsvSource({ "directorName1, mode1, page1, expectedResult1", "directorName2, mode2, page2, expectedResult2" })
    public void testDirectorSearch(String directorName, String mode, String page, ProductInfo expectedResult) {
        when(search.DirectorSearch(directorName, mode, page)).thenReturn(expectedResult);
        ProductInfo result = a4j.DirectorSearch(directorName, mode, page);
        assertEquals(expectedResult, result);
    }
}
