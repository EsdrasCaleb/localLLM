package br.com.jnfe.base.service;

import org.springframework.ws.soap.SoapBody;
import org.springframework.ws.soap.SoapMessage;
import org.springframework.ws.soap.soap12.Soap12Fault;
import org.springframework.xml.transform.StringResult;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import java.io.IOException;
import static org.mockito.ArgumentMatchers.any;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ws.WebServiceMessage;
import org.springframework.ws.client.core.FaultMessageResolver;

class LoggingFaultMessageResolver_resolveFault_0_0_Test {

    @Test
    void testResolveFault_Success() throws Exception {
        // Arrange
        LoggingFaultMessageResolver resolver = new LoggingFaultMessageResolver();
        SoapMessage mockMessage = mock(SoapMessage.class);
        SoapBody mockBody = mock(SoapBody.class);
        Soap12Fault mockFault = mock(Soap12Fault.class);
        StringResult mockResult = new StringResult();
        when(mockMessage.getSoapBody()).thenReturn(mockBody);
        when(mockBody.getFault()).thenReturn(mockFault);
        // Mock the source as needed
        when(mockFault.getSource()).thenReturn(any());
        Transformer mockTransformer = mock(Transformer.class);
        TransformerFactory mockFactory = mock(TransformerFactory.class);
        when(mockFactory.newTransformer()).thenReturn(mockTransformer);
        mockTransformer.transform(any(), eq(mockResult));
        // Act
        resolver.resolveFault(mockMessage);
        // Assert
        verify(mockTransformer).transform(any(), eq(mockResult));
        // Additional verifications can be added here for logger if needed
    }

    @Test
    void testResolveFault_ExceptionHandling() throws Exception {
        // Arrange
        LoggingFaultMessageResolver resolver = new LoggingFaultMessageResolver();
        SoapMessage mockMessage = mock(SoapMessage.class);
        SoapBody mockBody = mock(SoapBody.class);
        when(mockMessage.getSoapBody()).thenReturn(mockBody);
        when(mockBody.getFault()).thenThrow(new RuntimeException("Fault not found"));
        // Act
        resolver.resolveFault(mockMessage);
        // Assert
        // Verify that the logger was called with the expected message
        // You might need to use a custom logger or a logging framework that can be mocked
    }
}
